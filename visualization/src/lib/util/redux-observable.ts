import {ActionsObservable as ObservableOriginal} from 'redux-observable';
import {Action, ActionCreator, isType} from 'typescript-fsa';
import Redux from 'redux';
import 'rxjs/add/operator/filter';

declare module 'redux-observable' {
    interface ActionsObservable<T extends Redux.Action> {
        ofAction<T, P>(action: ActionCreator<P>): ActionsObservable<Action<P>>;
    }
}

ObservableOriginal.prototype.ofAction =
function <P>(this: ObservableOriginal<Action<P>>, actionCreater: ActionCreator<P>): ObservableOriginal<Action<P>> {
    return this.filter(action => (isType(action, actionCreater))) as ObservableOriginal<Action<P>>;
};
