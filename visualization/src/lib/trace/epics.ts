import {combineEpics, ActionsObservable, Epic} from 'redux-observable';
import {loadFile} from './actions';
import 'rxjs/add/operator/map';
import 'typescript-fsa-redux-observable';
import {Action} from "typescript-fsa";


const startFileLoad = (action$: ActionsObservable<Action<File>>) =>
    action$
        .ofAction(loadFile.started)
        .map(x => x.payload);

export const traceEpics = combineEpics();
