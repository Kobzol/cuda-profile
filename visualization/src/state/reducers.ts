import {parseReducer, TraceState} from "../lib/trace/reducer";

export interface AppState
{
    trace: TraceState,
    router: any;
}

export const reducers = {
    trace: parseReducer
};
