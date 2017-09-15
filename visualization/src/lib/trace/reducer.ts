import {TraceFile} from "./trace-file";
import {reducerWithInitialState} from "typescript-fsa-reducers";
import {loadFile} from "./actions";

export interface TraceState
{
    files: TraceFile[];
    fileId: number;
}

const reducer = reducerWithInitialState({
    files: [],
    fileId: 1
})
.case(loadFile.started, (state, payload) =>
{
    return {
        files: [...state.files, {
            id: state.fileId,
            name: payload.name,
            loading: true
        }],
        fileId: state.fileId + 1
    };
});

export const parseReducer = reducer;
