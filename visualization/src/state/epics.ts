import {combineEpics} from "redux-observable";
import {traceEpics} from "../lib/trace/epics";

export const rootEpic = combineEpics(
    traceEpics
);
