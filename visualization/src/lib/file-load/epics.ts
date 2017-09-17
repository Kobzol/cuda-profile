import {combineEpics, Epic} from 'redux-observable';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import {Action, Success} from 'typescript-fsa';
import 'typescript-fsa-redux-observable';
import {Failure} from 'typescript-fsa/lib';
import {loadFile} from './actions';
import {parseAndValidateFile} from './api';
import {FileLoadData} from './trace-file';


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
