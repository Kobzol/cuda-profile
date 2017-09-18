import {combineEpics} from 'redux-observable';
import {fileLoadEpics} from '../lib/file-load/epics';
import {traceEpics} from '../lib/trace/epics';

export const rootEpic = combineEpics(
    fileLoadEpics,
    traceEpics
);
