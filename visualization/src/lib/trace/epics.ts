import {combineEpics, Epic} from 'redux-observable';
import {loadTraceFile} from './actions';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'typescript-fsa-redux-observable';
import {parseTraceFileJson} from "./api";
import {Action, Success} from "typescript-fsa";
import {Failure} from "typescript-fsa/lib";
import {Observable} from "rxjs/Observable";


const loadTraceFileEpic: Epic<Action<Success<File, {}> | Failure<File, {}>>, {}> = action$ =>
    action$
        .ofAction(loadTraceFile.started)
        .flatMap(action => {
            return parseTraceFileJson(action.payload)
                .map(trace =>
                    loadTraceFile.done({
                        params: action.payload,
                        result: trace
                    })
                ).catch(error =>
                    Observable.of(loadTraceFile.failed({
                        params: action.payload,
                        error: error.message
                    }))
                );
        });

export const traceEpics = combineEpics(loadTraceFileEpic);
