import React from 'react';
import {Route} from "react-router";
import {LoadTracePage} from "./components/trace-loader/trace-loader";
import {ConnectedRouter} from "react-router-redux";
import {store, history} from './state/store';
import {Provider} from "react-redux";
import {DashboardPage} from "./components/dashboard/dashboard-page";
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
                            <Link to={"/dashboard"}>Dashboard</Link>
                        </nav>
                        <div>
                            <Route path="/" exact component={LoadTracePage} />
                            <Route path="/dashboard" component={DashboardPage} />
                        </div>
                    </div>
                </ConnectedRouter>
            </Provider>
        );
    }
}
