import {createSelector} from 'reselect';
import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {getErrorId, Errors} from '../state/errors';
import {GlobalState} from '../state/reducers';
import {pushOrReplaceInArray, replaceInArray} from '../util/immutable';
import {loadFile, resetFiles} from './actions';
import {FileType, TraceFile} from './file';

export interface FileLoaderState
{
    files: TraceFile[];
}

const reducer = reducerWithInitialState<FileLoaderState>({
    files: [],
})
.case(resetFiles, (state) => ({
    ...state,
    files: []
}))
.case(loadFile.started, (state, payload) => ({
        ...state,
        files: pushOrReplaceInArray(state.files, ((file) => file.name === payload.name), {
            name: payload.name,
            loading: true,
            content: null,
            type: FileType.Unknown,
            error: 0,
        })
}))
.case(loadFile.done, (state, payload) => ({
        ...state,
        files: replaceInArray(state.files, (file) => file.name === payload.params.name, {
            loading: false,
            content: payload.result.content,
            type: payload.result.type
        })
}))
.case(loadFile.failed, (state, payload) => ({
        ...state,
        files: replaceInArray(state.files, (file) => file.name === payload.params.name, {
            loading: false,
            type: FileType.Invalid,
            error: getErrorId(payload.error),
        })
}));

export const validTraceFiles = createSelector(
    (state: GlobalState) => state.fileLoader,
    (state: FileLoaderState) =>
        state.files.filter((file) =>
            !file.loading &&
            (file.type !== FileType.Invalid && file.type !== FileType.Unknown) &&
            file.content !== null &&
            file.error === Errors.None,
        ));
export const loadingFiles = createSelector(
    (state: GlobalState) => state.fileLoader,
    (state: FileLoaderState) =>
        state.files.filter((file) => file.loading)
);

export const fileLoaderReducer = reducer;
