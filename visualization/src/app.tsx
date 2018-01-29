import React, {PureComponent} from 'react';
import {Provider} from 'react-redux';
import {ConnectedRouter} from 'react-router-redux';
import {history, persistor, store} from './lib/state/store';
import {PersistGate} from 'redux-persist/es/integration/react';
import {Content} from './components/content/content';

export class App extends PureComponent
{
    render()
    {
        return (
            <Provider store={store}>
                <PersistGate
                    persistor={persistor}
                    loading={'Loading...'}>
                    <ConnectedRouter history={history}>
                        <Content />
                    </ConnectedRouter>
                </PersistGate>
            </Provider>
        );
    }
}
