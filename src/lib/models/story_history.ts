import { applyPatch, compare, Operation } from "fast-json-patch";
import { deepCopy } from "../util/objects";
import { DeltaPair } from "./world_state";

export type StoryActor = "storywriter" | "player";

export interface StoryTurn {
    id: string;
    actor: StoryActor;
    text: string;
    created_at: Date;
}

export class StoryHistory {
    private turns: StoryTurn[];
    constructor(initial: StoryTurn[] = []) {
        this.turns = initial;
    }

    public getLastNTurns(n: number = 10) {
        return deepCopy(this.turns.slice(-n));
    }

    public getAllTurns() {
        return deepCopy(this.turns);
    }

    public addTurn(
        actor: StoryActor,
        text: string
    ): { delta: DeltaPair; id: string } {
        const id = crypto.randomUUID();
        const mutator = (draft: { turns: StoryTurn[] }) => {
            draft.turns.push({
                id,
                actor,
                text,
                created_at: new Date(),
            });
        };
        return {
            delta: this._createDeltaPair(mutator) as DeltaPair,
            id,
        };
    }

    public removeLastNTurns(n: number = 1) {
        const mutator = (draft: { turns: StoryTurn[] }) => {
            for (let i = 0; i < n; i++) {
                if (draft.turns.pop() == undefined) return false;
            }
        };

        return this._createDeltaPair(mutator, true);
    }

    public updateTurn(
        turn: string | number,
        updates: Partial<Omit<StoryTurn, "id">>
    ): DeltaPair | null {
        const mutator = (draft: { turns: StoryTurn[] }) => {
            const plotIndex =
                typeof turn === "number"
                    ? turn
                    : draft.turns.findIndex((p) => p.id === turn);
            if (plotIndex === -1) return false; // Indicate failure
            Object.assign(draft.turns[plotIndex], updates);
            return true; // Indicate success
        };

        return this._createDeltaPair(mutator, true);
    }

    public applyDelta(delta: Operation[]) {
        const fullObject = {
            turns: this.turns,
        };
        const patched = applyPatch(fullObject, delta, true, false).newDocument;
        this.turns = patched.turns;
    }

    private _createDeltaPair(
        mutator: (draft: { turns: StoryTurn[] }) => void | boolean,
        canFail: boolean = false
    ): DeltaPair | null {
        const oldCopy = {
            turns: this.turns,
        };
        const newCopy = deepCopy(oldCopy);
        const result = mutator(newCopy);
        if (canFail && !result) {
            return null;
        }

        const apply = compare(oldCopy, newCopy);
        const revert = compare(newCopy, oldCopy);
        this.turns = newCopy.turns;

        return { apply, revert };
    }
}
