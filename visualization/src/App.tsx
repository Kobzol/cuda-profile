import React from 'react';
import {Route} from "react-router";
import {TraceLoader} from "./components/trace-loader/trace-loader";
import {ConnectedRouter} from "react-router-redux";
import {store, history} from './state/store';
import {Provider} from "react-redux";
import {KernelLaunchSelector} from "./components/kernel-launch-selector/kernel-launch-selector";
import {Link} from "react-router-dom";

export class App extends React.Component
{
    render()
    {
        return (
            <Provider store={store}>
                <ConnectedRouter history={history}>
                    <div>
                        <nav>
                            <Link to={"/"}>Home</Link>
                        </nav>
                        <div>
                            <Route path="/" exact component={TraceLoader} />
                            <Route path="/kernel-launches" component={KernelLaunchSelector} />
                        </div>
                    </div>
                </ConnectedRouter>
            </Provider>
        );
    }
}
