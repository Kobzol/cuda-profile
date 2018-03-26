import createHistory from 'history/createBrowserHistory';
import {routerMiddleware, routerReducer} from 'react-router-redux';
import {applyMiddleware, combineReducers, compose, createStore} from 'redux';
import {createLogger} from 'redux-logger';
import {createEpicMiddleware} from 'redux-observable';
import thunk from 'redux-thunk';
import {rootEpic} from './epics';
import {reducers} from './reducers';
import {persistStore, persistReducer} from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import url from 'url';

export const history = createHistory({
    basename: url.parse(process.env.PUBLIC_URL).pathname
});

const composeEnhancers = window['__REDUX_DEVTOOLS_EXTENSION_COMPOSE__'] || compose;
const router = routerMiddleware(history);
const epic = createEpicMiddleware(rootEpic);
const logger = createLogger();

const tracePersist = {
    key: 'trace',
    storage
};
const profilePersist = {
    key: 'project',
    storage,
    whitelist: ['profile']
};

const rootReducer = combineReducers({
    ...reducers,
    trace: persistReducer(tracePersist, reducers.trace),
    profile: persistReducer(profilePersist, reducers.profile),
    router: routerReducer
});

export const store = createStore(
    rootReducer,
    composeEnhancers(applyMiddleware(router, epic, thunk, logger))
);
export const persistor = persistStore(store);
