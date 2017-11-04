import {RouterState} from 'react-router-redux';
import {fileLoaderReducer, FileLoaderState} from '../file-load/reducer';
import {traceReducer, TraceState} from '../trace/reducer';
import {profileReducer, ProfileState} from '../profile/reducer';

export interface AppState
{
    fileLoader: FileLoaderState;
    trace: TraceState;
    profile: ProfileState;
    router: RouterState;
}

export const reducers = {
    fileLoader: fileLoaderReducer,
    trace: traceReducer,
    profile: profileReducer
};
