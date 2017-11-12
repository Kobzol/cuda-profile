import React from 'react';
import {Provider} from 'react-redux';
import {Route} from 'react-router';
import {Link} from 'react-router-dom';
import {ConnectedRouter} from 'react-router-redux';
import {TraceVisualisation} from './components/trace-visualisation/trace-visualisation';
import {TraceLoader} from './components/trace-loader/trace-loader';
import {history, persistor, store} from './lib/state/store';
import {Routes} from './lib/nav/routes';
import {PersistGate} from 'redux-persist/es/integration/react';

import './App.scss';

export class App extends React.Component
{
    render()
    {
        return (
            <Provider store={store}>
                <PersistGate
                    persistor={persistor}
                    loading={'Loading...'}>
                    <ConnectedRouter history={history}>
                        <div className='app'>
                            <ul className='nav nav-pills'>
                                <li><Link to={'/'}>Home</Link></li>
                            </ul>
                            <div className='content'>
                                <Route path={Routes.Root} exact component={TraceLoader} />
                                <Route path={Routes.TraceVisualisation} component={TraceVisualisation} />
                            </div>
                        </div>
                    </ConnectedRouter>
                </PersistGate>
            </Provider>
        );
    }
}
