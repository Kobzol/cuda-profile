import {parseReducer, FileLoaderState} from "../lib/file-load/reducer";

export interface AppState
{
    trace: FileLoaderState,
    router: any;
}

export const reducers = {
    trace: parseReducer
};
