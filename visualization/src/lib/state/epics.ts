import {combineEpics} from 'redux-observable';
import {fileLoadEpics} from '../file-load/epics';
import {traceEpics} from '../trace/epics';

export const rootEpic = combineEpics(
    fileLoadEpics,
    traceEpics
);
