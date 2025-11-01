import { createSignal, For, Show, type Component } from "solid-js";
import {
    runContextBuilderDemo,
    runGameDirectorDemo,
    runMemoryBankDemo,
    runPlotCardDemo,
    runStoryTreeDemo,
    runWorldStateDemo,
} from "./component_demo";

export default function App() {
    // runMemoryBankDemo().catch(console.error);
    // runPlotCardDemo().catch(console.error);
    // runStoryTreeDemo().catch(console.error);
    // runWorldStateDemo().catch(console.warn);
    // runContextBuilderDemo().catch(console.warn);
    // runGameDirectorDemo().catch(console.warn);

    return <>Hello World</>;
}
