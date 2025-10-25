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

/**
 * Represents a tree-like structure of `StoryNode` objects.
 * Each node has an associated `StoryTurn`.
 *
 * @class
 * @example
 * // Create a new instance of the StoryTree class
 * const tree = new StoryTree();
 *
 * // Add a new node to the tree
 * const node = new StoryNode({
 *   id: "nodeId",
 *   parent: null,
 *   children: [],
 *   turn: null,
 * });
 * tree.addNode(node);
 *
 * // Get the root node of the tree
 * const rootNode = tree.getRootNode();
 *
 * // Get the path to a node
 * const path = tree.getPathToNode("nodeId");
 *
 * // Get the depth of a node
 * const depth = tree.getDepth("nodeId");
 *
 * // Get the recent turns of a node
 * const recentTurns = tree.getRecentTurns("nodeId");
 *
 * // Get the nodes at a specific turn number
 * const nodesAtTurn = tree.getNodesAtTurn(2);
 *
 * // Get the deepest node in the tree
 * const deepestNode = tree.getDeepestNode();
 *
 * // Serialize the tree into a SerializedStoryTree object
 * const serializedTree = tree.serialize();
 *
 * // Deserialize a SerializedStoryTree object into a StoryTree instance
 * const deserializedTree = StoryTree.deserialize(serializedTree);
 */
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
        // deepCopy is essential to avoid mutation before comparison
        const draftState = deepCopy(oldState);

        const result = mutator(draftState);

        if (canFail && !result) {
            return null; // The mutation failed or was aborted
        }

        const apply = compare(oldState, draftState);
        const revert = compare(draftState, oldState);

        // Apply the change to the *actual* state
        this._setFullState(draftState);

        return { apply, revert };
    }

    public applyDelta(delta: Operation[]) {
        const fullObject = this._getFullState();
        const patched = applyPatch(fullObject, delta, true, false).newDocument;
        this._setFullState(patched);
    }

    public addNode(node: StoryNode): DeltaPair | null {
        const mutator = (draft: StoryTreeState) => {
            if (draft.nodes.has(node.id)) {
                console.warn(
                    `Node with ID ${node.id} already exists. Overwriting.`
                );
            }
            if (!node.parentId && draft.rootNodeId) {
                console.error("Attempting to add a second root node.");
                return false; // Abort
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
                return false; // Node not found, fail
            }
            node.turn = newTurnData;
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
            // 1. Unlink from parent
            if (node.parentId) {
                const parent = draft.nodes.get(node.parentId);
                if (parent) {
                    parent.childrenIds = parent.childrenIds.filter(
                        (id) => id !== nodeId
                    );
                }
            }

            // 2. BFS/DFS to find all descendants and delete
            const stack: string[] = [nodeId];
            while (stack.length > 0) {
                const currentId = stack.pop()!;
                const currentNode = draft.nodes.get(currentId);

                if (currentNode) {
                    stack.push(...currentNode.childrenIds); // Add children
                    draft.nodes.delete(currentId);
                    deletedNodes.push(currentNode);
                }
            }
            return true;
        };

        const delta = this._createDeltaPair(mutator, true);

        if (!delta) return null; // Should not happen if root check passed

        // Return leafs-first so re-addition is parent-first
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
        const queue: [string, number][] = [[this.rootNodeId, 1]]; // [nodeId, depth]

        while (queue.length > 0) {
            const [currentId, currentDepth] = queue.shift()!;
            const node = this.nodes.get(currentId);
            if (!node) continue;

            if (currentDepth === turnNumber) {
                results.push(node);
                continue; // Don't traverse deeper
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
        const queue: [string, number][] = [[this.rootNodeId, 1]]; // [nodeId, depth]

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
