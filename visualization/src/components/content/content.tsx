import React, {PureComponent} from 'react';
import {TraceVisualisation} from '../trace-visualisation/trace-visualisation';
import {TraceLoader} from '../trace-loader/trace-loader';
import {Routes} from '../../lib/nav/routes';
import {Link} from 'react-router-dom';
import {Route, withRouter} from 'react-router';

import style from './content.scss';

class ContentComponent extends PureComponent
{
    render()
    {
        return (
            <div className={style.root}>
                <div className={style.header}>
                    <span className={style.appName}>Trace visualisation</span>
                    <ul className='nav nav-pills'>
                        <li><Link to={Routes.Root}>Load trace</Link></li>
                    </ul>
                </div>
                <div className={style.content}>
                    <Route path={Routes.Root} exact component={TraceLoader} />
                    <Route path={Routes.TraceVisualisation} component={TraceVisualisation} />
                </div>
            </div>
        );
    }
}

export const Content = withRouter(ContentComponent);
