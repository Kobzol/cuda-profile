import {RouterState} from 'react-router-redux';
import {parseReducer, FileLoaderState} from '../lib/file-load/reducer';

export interface AppState
{
    trace: FileLoaderState;
    router: RouterState;
}

export const reducers = {
    trace: parseReducer
};
