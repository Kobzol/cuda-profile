import {createSelector} from 'reselect';
import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {getErrorId, Errors} from '../../state/errors';
import {AppState} from '../../state/reducers';
import {pushOrReplaceArray, replaceArray} from '../util/immutable';
import {loadFile} from './actions';
import {FileType, TraceFile} from './trace-file';

export interface FileLoaderState
{
    files: TraceFile[];
}

const reducer = reducerWithInitialState({
    files: [],
})
.case(loadFile.started, (state: FileLoaderState, payload) => {
    return {
        ...state,
        files: pushOrReplaceArray(state.files, ((file) => file.name === payload.name), {
            name: payload.name,
            loading: true,
            content: null,
            type: FileType.Unknown,
            error: 0,
        }),
    };
})
.case(loadFile.done, (state: FileLoaderState, payload) => {
    return {
        ...state,
        files: replaceArray(state.files, (file) => file.name === payload.params.name, {
                loading: false,
                content: payload.result,
            }),
    };
})
.case(loadFile.failed, (state: FileLoaderState, payload) => {
    return {
        ...state,
        files: replaceArray(state.files, (file) => file.name === payload.params.name, {
            loading: false,
            type: FileType.Invalid,
            error: getErrorId(payload.error),
        }),
    };
});

export const validTraceFiles = createSelector(
    (state: AppState) => state.trace,
    (state: FileLoaderState) =>
        state.files.filter((file) =>
            !file.loading &&
            (file.type !== FileType.Invalid && file.type !== FileType.Unknown) &&
            file.content !== null &&
            file.error === Errors.None,
        ));

export const parseReducer = reducer;
