import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {buildProfile, selectTrace} from '../../lib/trace/actions';
import {AppState} from '../../lib/state/reducers';
import {KernelTimeline} from '../kernel-timeline/kernel-timeline';
import {Profile} from '../../lib/trace/profile';
import {TraceSelection} from '../../lib/trace/trace-selection';
import {Kernel} from '../../lib/trace/kernel';
import {Trace} from '../../lib/trace/trace';
import {selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {AccessTimeline} from '../access-timeline/access-timeline';
import {ToggleWrapper} from '../toggle-wrapper/toggle-wrapper';

import './trace-visualisation.css';

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
    selectedKernel: Kernel;
    selectedTrace: Trace;
}
interface DispatchProps
{
    buildProfile: (files: TraceFile[]) => {};
    selectTrace: (selection: TraceSelection) => {};
}

type Props = StateProps & DispatchProps;

interface State
{
    showKernelTimeline: boolean;
}

class TraceVisualisationComponent extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            showKernelTimeline: true
        };
    }

    componentWillMount()
    {
        this.props.buildProfile(this.props.files);
    }
    componentWillReceiveProps(nextProps: Props)
    {
        if (this.props.selectedTrace === null && nextProps.selectedTrace !== null)
        {
            this.hideKernelTimeline();
        }
    }

    render()
    {
        if (this.props.profile === null)
        {
            return this.renderLoading();
        }
        else return this.renderContent();
    }

    renderLoading = (): JSX.Element =>
    {
        return (<div>Loading profile data...</div>);
    }

    renderContent = (): JSX.Element =>
    {
        return (
            <div className='trace-visualisation'>
                {this.renderKernelTimeline()}
                {this.props.selectedTrace !== null &&
                    <div className='access-timeline'>
                        <AccessTimeline
                            kernel={this.props.selectedKernel}
                            trace={this.props.selectedTrace} />
                    </div>
                }
            </div>
        );
    }

    renderKernelTimeline = (): JSX.Element =>
    {
        return (
            <ToggleWrapper
                onShow={this.showKernelTimeline}
                showContent={this.state.showKernelTimeline}
                toggleText={'Select trace'}>
                <KernelTimeline
                    selectTrace={this.props.selectTrace}
                    profile={this.props.profile} />
            </ToggleWrapper>
        );
    }

    showKernelTimeline = () =>
    {
        this.setState({
            showKernelTimeline: true
        });
    }
    hideKernelTimeline = () =>
    {
        this.setState({
            showKernelTimeline: false
        });
    }
}

export const TraceVisualisation = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.fileLoader.files,
    profile: state.trace.profile,
    selectedKernel: selectedKernel(state),
    selectedTrace: selectedTrace(state)
}), {
    buildProfile: buildProfile.started,
    selectTrace: selectTrace
})(TraceVisualisationComponent);
