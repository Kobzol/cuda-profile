import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {buildProfile, selectAccessGroup, selectTrace} from '../../lib/trace/actions';
import {AppState} from '../../lib/state/reducers';
import {KernelTimeline} from './kernel-timeline/kernel-timeline';
import {Profile} from '../../lib/profile/profile';
import {TraceSelection} from '../../lib/trace/trace-selection';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {selectedAccessGroup, selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {AccessTimeline} from './access-timeline/access-timeline';
import {ToggleWrapper} from '../toggle-wrapper/toggle-wrapper';
import {MemoryMap} from './memory-map/memory-map';
import {MemoryAccessGroup} from '../../lib/profile/memory-access';

import './trace-visualisation.css';
import {TraceAccess} from './trace-access/trace-access';

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
    selectedKernel: Kernel;
    selectedTrace: Trace;
    selectedAccessGroup: MemoryAccessGroup;
}
interface DispatchProps
{
    buildProfile: (files: TraceFile[]) => {};
    selectTrace: (selection: TraceSelection) => {};
    selectAccessGroup: (index: number) => {};
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
        let content: JSX.Element = null;
        if (this.props.selectedTrace !== null)
        {
            content = this.renderTraceContent(
                this.props.selectedKernel,
                this.props.selectedTrace,
                this.props.selectedAccessGroup
            );
        }
        return (
            <div className='trace-visualisation'>
                {this.renderKernelTimeline()}
                {content}
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

    renderTraceContent = (kernel: Kernel, trace: Trace, accessGroup: MemoryAccessGroup): JSX.Element =>
    {
        return (
            <div>
                {accessGroup !== null &&
                    <TraceAccess
                        trace={trace}
                        accessGroup={accessGroup} />}
                {this.renderAccessTimeline(kernel, trace)}
            </div>
        );
    }
    renderAccessTimeline = (kernel: Kernel, trace: Trace): JSX.Element =>
    {
        return (
            <div className='access-timeline'>
                <AccessTimeline
                    kernel={kernel}
                    trace={trace}
                    selectAccessGroup={this.props.selectAccessGroup} />
            </div>
        );
    }

    showKernelTimeline = () =>
    {
        this.setState(() => ({
            showKernelTimeline: true
        }));
    }
    hideKernelTimeline = () =>
    {
        this.setState(() => ({
            showKernelTimeline: false
        }));
    }
}

export const TraceVisualisation = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.fileLoader.files,
    profile: state.trace.profile,
    selectedKernel: selectedKernel(state),
    selectedTrace: selectedTrace(state),
    selectedAccessGroup: selectedAccessGroup(state)
}), {
    buildProfile: buildProfile.started,
    selectTrace: selectTrace,
    selectAccessGroup: selectAccessGroup
})(TraceVisualisationComponent);
