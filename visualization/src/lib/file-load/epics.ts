import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import '../util/redux-observable';
import {loadFile} from './actions';
import {parseAndValidateFile} from './file';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
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

export const fileLoadEpics = combineEpics(loadTraceFileEpic);
