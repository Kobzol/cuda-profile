import {RouterState} from 'react-router-redux';
import {fileLoaderReducer, FileLoaderState} from '../lib/file-load/reducer';
import {traceReducer, TraceState} from '../lib/trace/reducer';

export interface AppState
{
    fileLoader: FileLoaderState;
    trace: TraceState;
    router: RouterState;
}

export const reducers = {
    fileLoader: fileLoaderReducer,
    trace: traceReducer
};
