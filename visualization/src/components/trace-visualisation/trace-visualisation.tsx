import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {buildProfile, selectWarps, selectTrace} from '../../lib/trace/actions';
import {AppState} from '../../lib/state/reducers';
import {KernelTimeline} from './kernel-timeline/kernel-timeline';
import {Profile} from '../../lib/profile/profile';
import {TraceSelection} from '../../lib/trace/selection';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {selectedWarps, selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {WarpTimeline} from './warp-timeline/warp-timeline';
import {ToggleWrapper} from '../toggle-wrapper/toggle-wrapper';
import {Warp} from '../../lib/profile/warp';

import './trace-visualisation.css';
import {WarpList} from './warp-list/warp-list';
import {MemoryBlock} from './memory-block/memory-block';
import {WarpAddressSelection} from './selection';

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
    selectedKernel: Kernel;
    selectedTrace: Trace;
    selectedWarps: Warp[];
}
interface DispatchProps
{
    buildProfile: (files: TraceFile[]) => {};
    selectTrace: (selection: TraceSelection) => {};
    selectWarps: (warps: number[]) => {};
}

type Props = StateProps & DispatchProps;

interface State
{
    showKernelTimeline: boolean;
    rangeSelections: WarpAddressSelection[];
}

class TraceVisualisationComponent extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            showKernelTimeline: true,
            rangeSelections: []
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
                this.props.selectedWarps
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

    renderTraceContent = (kernel: Kernel, trace: Trace, warps: Warp[]): JSX.Element =>
    {
        return (
            <div>
                <div className='trace-wrapper'>
                    <WarpList
                        trace={trace}
                        warps={warps}
                        selectRange={(range) => this.setState({
                            rangeSelections: range === null ? [] : [range]
                        })} />
                    <div>
                        {trace.allocations.slice(-1).map(alloc =>
                            <MemoryBlock
                                key={alloc.address}
                                allocation={alloc}
                                rangeSelections={this.state.rangeSelections} />
                        )}
                    </div>
                </div>
                {this.renderAccessTimeline(kernel, trace)}
            </div>
        );
    }
    renderAccessTimeline = (kernel: Kernel, trace: Trace): JSX.Element =>
    {
        return (
            <div className='warp-timeline'>
                <WarpTimeline
                    kernel={kernel}
                    trace={trace}
                    selectWarps={this.props.selectWarps} />
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
    selectedWarps: selectedWarps(state)
}), {
    buildProfile: buildProfile.started,
    selectTrace: selectTrace,
    selectWarps: selectWarps
})(TraceVisualisationComponent);
