import {TraceFile} from "./trace-file";
import {reducerWithInitialState} from "typescript-fsa-reducers";
import {loadTraceFile} from "./actions";
import {replaceArray} from "../util/immutable";
import {createSelector} from "reselect";
import {AppState} from "../../state/reducers";

export interface TraceState
{
    files: TraceFile[];
    fileId: number;
}

const reducer = reducerWithInitialState({
    files: [],
    fileId: 1
})
.case(loadTraceFile.started, (state: TraceState, payload) => {
    return {
        ...state,
        files: [...state.files, {
            id: state.fileId,
            name: payload.name,
            loading: true,
            content: {},
            error: null
        }],
        fileId: state.fileId + 1
    };
})
.case(loadTraceFile.done, (state: TraceState, payload) => {
    return {
        ...state,
        files: replaceArray(state.files, file => file.name === payload.params.name, {
                loading: false,
                content: payload.result
            })
    };
})
.case(loadTraceFile.failed, (state: TraceState, payload) => {
    return {
        ...state,
        files: replaceArray(state.files, file => file.name === payload.params.name, {
            loading: false,
            error: payload.error
        })
    }
});

export const validTraceFiles = createSelector(
    (state: AppState) => state.trace,
    (state: TraceState) =>
        state.files.filter(file =>
            !file.loading &&
            file.error === null &&
            file.content !== {}
        ));

export const parseReducer = reducer;
