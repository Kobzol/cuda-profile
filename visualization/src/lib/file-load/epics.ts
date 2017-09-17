import {combineEpics, Epic} from 'redux-observable';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'typescript-fsa-redux-observable';
import {parseAndValidateFile} from "./api";
import {Action, Success} from "typescript-fsa";
import {Failure} from "typescript-fsa/lib";
import {Observable} from "rxjs/Observable";
import {FileLoadData} from "./trace-file";
import {loadFile} from "./actions";


const loadTraceFileEpic: Epic<Action<Success<File, FileLoadData> | Failure<File, Error>>, {}> = action$ =>
    action$
        .ofAction(loadFile.started)
        .flatMap(action => {
            return parseAndValidateFile(action.payload)
                .map(loadData =>
                    loadFile.done({
                        params: action.payload,
                        result: loadData
                    })
                ).catch(error =>
                    Observable.of(loadFile.failed({
                        params: action.payload,
                        error
                    }))
                );
        });

export const traceEpics = combineEpics(loadTraceFileEpic);
