import {combineEpics} from "redux-observable";
import {traceEpics} from "../lib/file-load/epics";

export const rootEpic = combineEpics(
    traceEpics
);
