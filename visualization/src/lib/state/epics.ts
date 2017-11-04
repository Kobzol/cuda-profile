import {combineEpics} from 'redux-observable';
import {fileLoadEpics} from '../file-load/epics';
import {traceEpics} from '../trace/epics';
import {profileEpics} from '../profile/epics';

export const rootEpic = combineEpics(
    fileLoadEpics,
    traceEpics,
    profileEpics
);
