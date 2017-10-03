import React from 'react';
import {Provider} from 'react-redux';
import {Route} from 'react-router';
import {Link} from 'react-router-dom';
import {ConnectedRouter} from 'react-router-redux';
import {TraceVisualisation} from './components/trace-visualisation/trace-visualisation';
import {TraceLoader} from './components/trace-loader/trace-loader';
import {history, store} from './lib/state/store';
import {Routes} from './lib/nav/routes';

import './App.css';

export class App extends React.Component
{
    render()
    {
        return (
            <Provider store={store}>
                <ConnectedRouter history={history}>
                    <div className='app'>
                        <nav className='nav'>
                            <Link to={'/'}>Home</Link>
                        </nav>
                        <div className='content'>
                            <Route path='/' exact component={TraceLoader} />
                            <Route path={Routes.TraceVisualisation} component={TraceVisualisation} />
                        </div>
                    </div>
                </ConnectedRouter>
            </Provider>
        );
    }
}
