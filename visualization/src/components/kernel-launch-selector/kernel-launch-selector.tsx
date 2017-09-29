import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {buildProfile, selectTrace} from '../../lib/trace/actions';
import {AppState} from '../../state/reducers';
import {KernelTimeline} from '../kernel-timeline/kernel-timeline';
import {Profile} from '../../lib/trace/profile';
import {TraceSelection} from '../../lib/trace/trace-selection';

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
}
interface DispatchProps
{
    buildProfile: (files: TraceFile[]) => {};
    selectTrace: (selection: TraceSelection) => {};
}

class KernelLaunchSelectorComponent extends PureComponent<StateProps & DispatchProps>
{
    componentWillMount()
    {
        this.props.buildProfile(this.props.files);
    }

    render()
    {
        if (this.props.profile === null)
        {
            return (<div>Loading profile...</div>);
        }
        else return (
            <KernelTimeline
                profile={this.props.profile}
                selectTrace={this.props.selectTrace} />
        );
    }
}

export const KernelLaunchSelector = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.fileLoader.files,
    profile: state.trace.profile
}), {
    buildProfile: buildProfile.started,
    selectTrace: selectTrace
})(KernelLaunchSelectorComponent);
