import React from 'react';
import {Provider} from 'react-redux';
import {Route} from 'react-router';
import {Link} from 'react-router-dom';
import {ConnectedRouter} from 'react-router-redux';
import {KernelLaunchSelector} from './components/kernel-launch-selector/kernel-launch-selector';
import {TraceLoader} from './components/trace-loader/trace-loader';
import {history, store} from './state/store';
import './App.css';

export class App extends React.Component
{
    render()
    {
        return (
            <Provider store={store}>
                <ConnectedRouter history={history}>
                    <div>
                        <nav>
                            <Link to={'/'}>Home</Link>
                        </nav>
                        <div>
                            <Route path='/' exact component={TraceLoader} />
                            <Route path='/kernel-timeline' component={KernelLaunchSelector} />
                        </div>
                    </div>
                </ConnectedRouter>
            </Provider>
        );
    }
}
