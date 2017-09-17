import React, {PureComponent} from 'react';
import {connect} from 'react-redux';

class KernelLaunchSelectorComponent extends PureComponent
{
    render()
    {
        return (
            <div>dashboard</div>
        );
    }
}

export const KernelLaunchSelector = connect()(KernelLaunchSelectorComponent);
