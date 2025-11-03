import { applyPatch, compare, Operation } from "fast-json-patch";
import { deepCopy } from "../util/objects";
import { DeltaPair } from "./world_state";

export type StoryActor = "storywriter" | "player";

export interface StoryTurn {
    actor: StoryActor;
    text: string;
    directorThinking?: string;
}

export interface StoryNode {
    id: string;
    parentId: string;
    childrenIds: string[];
    turn: StoryTurn;
    deltas: DeltaPair[];
}

export interface SerializedStoryTree {
    rootNodeId: string | null;
    nodes: [string, StoryNode][];
}

type StoryTreeState = {
    nodes: Map<string, StoryNode>;
    rootNodeId: string | null;
};

export class StoryTree {
    private nodes: Map<string, StoryNode> = new Map();
    private rootNodeId: string | null = null;

    constructor(initialState?: StoryTreeState) {
        if (initialState) {
            this.nodes = initialState.nodes;
            this.rootNodeId = initialState.rootNodeId;
        }
    }

    private _getFullState(): StoryTreeState {
        return {
            nodes: this.nodes,
            rootNodeId: this.rootNodeId,
        };
    }

    private _setFullState(state: StoryTreeState) {
        this.nodes = state.nodes;
        this.rootNodeId = state.rootNodeId;
    }

    private _createDeltaPair(
        mutator: (draftState: StoryTreeState) => boolean | void,
        canFail: boolean = false
    ): DeltaPair | null {
        const oldState = this._getFullState();
        const draftState = deepCopy(oldState);

        const result = mutator(draftState);

        if (canFail && !result) {
            return null;
        }

        const jsonOldState = {
            ...oldState,
            nodes: Object.fromEntries(oldState.nodes),
        };
        const jsonDraftState = {
            ...draftState,
            nodes: Object.fromEntries(draftState.nodes),
        };

        const apply = compare(jsonOldState, jsonDraftState);
        const revert = compare(jsonDraftState, jsonOldState);

        this._setFullState(draftState);

        return { apply, revert };
    }

    public applyDelta(delta: Operation[]) {
        const fullObject = this._getFullState();
        const jsonFullObject = {
            ...fullObject,
            nodes: Object.fromEntries(fullObject.nodes),
        };
        const patched = applyPatch(
            jsonFullObject,
            delta,
            true,
            false
        ).newDocument;
        this._setFullState({
            ...patched,
            nodes: new Map(Object.entries(patched.nodes)),
        });
    }

    public addNode(node: StoryNode): DeltaPair | null {
        const mutator = (draft: StoryTreeState) => {
            console.log(draft.nodes);
            if (draft.nodes.has(node.id)) {
                console.warn(
                    `Node with ID ${node.id} already exists. Overwriting.`
                );
            }
            if (!node.parentId && draft.rootNodeId) {
                console.error("Attempting to add a second root node.");
                return false;
            }

            if (!node.parentId && !draft.rootNodeId) {
                draft.rootNodeId = node.id;
            }

            draft.nodes.set(node.id, node);

            if (node.parentId) {
                const parent = draft.nodes.get(node.parentId);
                if (parent && !parent.childrenIds.includes(node.id)) {
                    parent.childrenIds.push(node.id);
                }
            }
            return true;
        };

        return this._createDeltaPair(mutator, true);
    }

    public editNode(nodeId: string, newTurnData: StoryTurn): DeltaPair | null {
        const mutator = (draft: StoryTreeState) => {
            const node = draft.nodes.get(nodeId);
            if (!node) {
                return false;
            }
            node.turn = newTurnData;
            return true;
        };

        return this._createDeltaPair(mutator, true);
    }

    public updateNode(
        nodeId: string,
        newTurn: StoryTurn,
        newDeltas: DeltaPair[]
    ): DeltaPair | null {
        const mutator = (draft: StoryTreeState) => {
            const node = draft.nodes.get(nodeId);
            if (!node) {
                return false;
            }
            node.turn = newTurn;
            node.deltas = newDeltas;
            return true;
        };

        return this._createDeltaPair(mutator, true);
    }

    public deleteBranch(
        nodeId: string
    ): { deletedNodes: StoryNode[]; delta: DeltaPair } | null {
        const node = this.nodes.get(nodeId);
        if (!node || node.id === this.rootNodeId) {
            console.error("Cannot delete node: not found or is root node.");
            return null;
        }

        const deletedNodes: StoryNode[] = [];

        const mutator = (draft: StoryTreeState) => {
            if (node.parentId) {
                const parent = draft.nodes.get(node.parentId);
                if (parent) {
                    parent.childrenIds = parent.childrenIds.filter(
                        (id) => id !== nodeId
                    );
                }
            }

            const stack: string[] = [nodeId];
            while (stack.length > 0) {
                const currentId = stack.pop()!;
                const currentNode = draft.nodes.get(currentId);

                if (currentNode) {
                    stack.push(...currentNode.childrenIds);
                    draft.nodes.delete(currentId);
                    deletedNodes.push(currentNode);
                }
            }
            return true;
        };

        const delta = this._createDeltaPair(mutator, true);

        if (!delta) return null;

        return { deletedNodes: deletedNodes.reverse(), delta };
    }

    public getNode(nodeId: string): StoryNode | undefined {
        return this.nodes.get(nodeId);
    }

    public getRootNode(): StoryNode | undefined {
        return this.rootNodeId ? this.nodes.get(this.rootNodeId) : undefined;
    }

    public getPathToNode(nodeId: string): StoryNode[] {
        const path: StoryNode[] = [];
        let current = this.nodes.get(nodeId);

        while (current) {
            path.push(current);
            if (!current.parentId) break;
            current = this.nodes.get(current.parentId);
        }

        return path.reverse();
    }

    public getDepth(nodeId: string): number {
        return this.getPathToNode(nodeId).length;
    }

    public getRecentTurns(nodeId: string, n: number): StoryNode["turn"][] {
        const path = this.getPathToNode(nodeId);
        return path.slice(-n).map((node) => node.turn);
    }

    public getNodesAtTurn(turnNumber: number): StoryNode[] {
        if (turnNumber < 1 || !this.rootNodeId) return [];

        const results: StoryNode[] = [];
        const queue: [string, number][] = [[this.rootNodeId, 1]];

        while (queue.length > 0) {
            const [currentId, currentDepth] = queue.shift()!;
            const node = this.nodes.get(currentId);
            if (!node) continue;

            if (currentDepth === turnNumber) {
                results.push(node);
                continue;
            }

            if (currentDepth < turnNumber) {
                for (const childId of node.childrenIds) {
                    queue.push([childId, currentDepth + 1]);
                }
            }
        }
        return results;
    }

    public getDeepestNode(): StoryNode | null {
        if (!this.rootNodeId) return null;

        let deepestNode: StoryNode | null =
            this.nodes.get(this.rootNodeId) || null;
        let maxDepth = 1;
        const queue: [string, number][] = [[this.rootNodeId, 1]];

        while (queue.length > 0) {
            const [currentId, currentDepth] = queue.shift()!;
            const currentNode = this.nodes.get(currentId);
            if (!currentNode) continue;

            if (currentDepth > maxDepth) {
                maxDepth = currentDepth;
                deepestNode = currentNode;
            }

            for (const childId of currentNode.childrenIds) {
                queue.push([childId, currentDepth + 1]);
            }
        }
        return deepestNode;
    }

    public serialize(): SerializedStoryTree {
        return {
            nodes: Array.from(this.nodes.entries()),
            rootNodeId: this.rootNodeId,
        };
    }

    public static deserialize(data: SerializedStoryTree): StoryTree {
        const state: StoryTreeState = {
            nodes: new Map(data.nodes),
            rootNodeId: data.rootNodeId,
        };
        return new StoryTree(state);
    }
}
