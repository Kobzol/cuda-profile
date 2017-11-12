import createHistory from 'history/createBrowserHistory';
import {routerMiddleware, routerReducer} from 'react-router-redux';
import {applyMiddleware, compose, createStore} from 'redux';
import {createLogger} from 'redux-logger';
import {createEpicMiddleware} from 'redux-observable';
import {rootEpic} from './epics';
import {reducers} from './reducers';
import {persistStore, persistCombineReducers} from 'redux-persist';
import storage from 'redux-persist/es/storage';
import url from 'url';

export const history = createHistory({
    basename: url.parse(process.env.PUBLIC_URL).pathname
});

const composeEnhancers = window['__REDUX_DEVTOOLS_EXTENSION_COMPOSE__'] || compose;
const router = routerMiddleware(history);
const epic = createEpicMiddleware(rootEpic);
const logger = createLogger();
const persistConfig = {
    key: 'root',
    storage,
    whitelist: ['trace', 'profile']
};

export const store = createStore(
    persistCombineReducers(persistConfig, {
        ...reducers,
        router: routerReducer
    }),
    composeEnhancers(applyMiddleware(router, epic, logger))
);
export const persistor = persistStore(store);
