import { compare, applyPatch, Operation } from "fast-json-patch";
import { deepCopy, deepMerge } from "../util/objects";

export interface Plot {
    id: string;
    title: string;
    description: string;
    player_alignment: number;
    created_at: Date;
}

export interface DeltaPair {
    apply: Operation[];
    revert: Operation[];
}

export class WorldState {
    private state: { [key: string]: any };
    private plots: Plot[];

    constructor(initialState: object = {}, initialPlots: Plot[] = []) {
        this.plots = deepCopy(initialPlots);
        this.state = deepCopy(initialState);
    }

    public getState() {
        return deepCopy(this.state);
    }

    public getPlots() {
        return deepCopy(this.plots);
    }

    public toText(): string {
        let contextString = "--- World State ---\n";
        contextString += JSON.stringify(this.state, null, 2);
        contextString += "\n\n--- Active Plotlines ---\n";

        if (this.plots.length == 0) {
            contextString += "None.\n";
        } else {
            this.plots.forEach((plot) => {
                contextString += `- ${plot.title}: ${
                    plot.description
                } (Player Alignment: ${plot.player_alignment.toFixed(2)})\n`;
            });
        }
        return contextString;
    }

    public deepSet(path: string, value: any): DeltaPair {
        const mutator = (draft: {
            state: { [key: string]: any };
            plots: Plot[];
        }) => {
            const keys = path.split("/");
            let current = draft.state;
            for (let i = 0; i < keys.length - 1; i++) {
                current = current[keys[i]] = current[keys[i]] || {};
            }
            current[keys[keys.length - 1]] = value;
        };
        return this._createDeltaPair(mutator) as DeltaPair;
    }

    public patchState(partialState: any): DeltaPair {
        const mutator = (draft: { state: any; plots: Plot[] }) => {
            deepMerge(draft.state, partialState);
        };
        return this._createDeltaPair(mutator) as DeltaPair;
    }

    public addPlot(newPlot: Omit<Plot, "id" | "created_at">): {
        newId: string;
        delta: DeltaPair;
    } {
        const id = crypto.randomUUID();
        const mutator = (draft: { state: any; plots: Plot[] }) => {
            draft.plots.push({
                ...newPlot,
                id,
                created_at: new Date(),
            });
        };

        const delta = this._createDeltaPair(mutator) as DeltaPair;
        return { newId: id, delta };
    }

    public updatePlot(
        plotId: string,
        updates: Partial<Omit<Plot, "id">>
    ): DeltaPair | null {
        const mutator = (draft: { state: any; plots: Plot[] }) => {
            const plotIndex = draft.plots.findIndex((p) => p.id === plotId);
            if (plotIndex === -1) return false; // Indicate failure
            Object.assign(draft.plots[plotIndex], updates);
            return true; // Indicate success
        };

        return this._createDeltaPair(mutator, true);
    }

    public removePlot(plotId: string): DeltaPair | null {
        const mutator = (draft: { state: any; plots: Plot[] }) => {
            const initialLength = draft.plots.length;
            draft.plots = draft.plots.filter((p) => p.id !== plotId);
            return draft.plots.length < initialLength; // Return success/failure
        };

        return this._createDeltaPair(mutator, true);
    }

    public applyDelta(delta: Operation[]) {
        const fullObject = {
            state: this.state,
            plots: this.plots,
        };
        const patched = applyPatch(fullObject, delta, true, false).newDocument;
        this.state = patched.state;
        this.plots = patched.plots;
    }

    private _createDeltaPair(
        mutator: (draft: any) => boolean | void,
        canFail: boolean = false
    ): DeltaPair | null {
        const oldFullObject = {
            state: this.state,
            plots: this.plots,
        };
        const newFullObject = deepCopy(oldFullObject);

        const result = mutator(newFullObject);
        if (canFail && !result) {
            return null;
        }

        const apply = compare(oldFullObject, newFullObject);
        const revert = compare(newFullObject, oldFullObject);
        this.state = newFullObject.state;
        this.plots = newFullObject.plots;

        return { apply, revert };
    }
}

const ws = new WorldState();
ws.addPlot({
    title: "Save the king!",
    description:
        "The player hears of the threat to the king. They venture forth to save him and keep the kingdom safe!",
    player_alignment: 0.9,
});
