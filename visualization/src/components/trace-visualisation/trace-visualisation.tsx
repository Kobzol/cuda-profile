import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {selectWarps, selectTrace} from '../../lib/trace/actions';
import {GlobalState} from '../../lib/state/reducers';
import {TraceTimeline} from './trace-timeline/trace-timeline';
import {Profile} from '../../lib/profile/profile';
import {AddressRange, TraceSelection} from '../../lib/trace/selection';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {selectedWarps, selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {WarpTimeline} from './warp-timeline/warp-timeline';
import {Warp} from '../../lib/profile/warp';
import {WarpList} from './warp-list/warp-list';
import {WarpAddressSelection} from '../../lib/trace/selection';
import {Routes} from '../../lib/nav/routes';
import {push} from 'react-router-redux';
import {MemoryList} from './memory-list/memory-list';

import './trace-visualisation.scss';
import {Button, Glyphicon} from 'react-bootstrap';
import * as moment from 'moment';

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
    selectedKernel: Kernel;
    selectedTrace: Trace;
    selectedWarps: Warp[];
    traceSelection: TraceSelection;
    warpSelection: number[];
}
interface DispatchProps
{
    selectTrace: (selection: TraceSelection) => {};
    selectWarps: (warps: number[]) => {};
    goToPage: (page: string) => {};
}

type Props = StateProps & DispatchProps;

interface State
{
    rangeSelections: WarpAddressSelection[];
    memorySelection: AddressRange | null;
}

class TraceVisualisationComponent extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            rangeSelections: [],
            memorySelection: null
        };
    }

    componentWillMount()
    {
        if (this.props.profile === null)
        {
            this.props.goToPage(Routes.Root);
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
                {this.renderTraceTimeline()}
                {content}
            </div>
        );
    }
    renderTraceTimeline = (): JSX.Element =>
    {
        if (this.props.selectedKernel !== null)
        {
            const start = moment(this.props.selectedTrace.start).format('HH:mm:ss.SSS');
            const end = moment(this.props.selectedTrace.end).format('HH:mm:ss.SSS');

            return (
                <div className='kernel-details'>
                    <div>
                        <h3>
                            Selected trace
                        </h3>
                        <div>{`${this.props.selectedKernel.name} from ${start} to ${end}`}</div>
                    </div>
                    <Button
                        className='kernel-deselect'
                        onClick={this.deselectTrace}
                        bsStyle='primary'>
                        <Glyphicon glyph='list' /> Select another trace
                    </Button>
                </div>
            );
        }
        else
        {
            return (
                <TraceTimeline
                    selectTrace={this.props.selectTrace}
                    profile={this.props.profile}
                    selection={this.props.traceSelection} />
            );
        }
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
                        })}
                        memorySelection={this.state.memorySelection} />
                    <MemoryList
                        allocations={this.props.selectedTrace.allocations}
                        rangeSelections={this.state.rangeSelections}
                        onMemorySelect={this.setMemorySelection} />
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
                    selectWarps={this.props.selectWarps}
                    selectedWraps={this.props.warpSelection} />
            </div>
        );
    }

    setMemorySelection = (memorySelection: AddressRange) =>
    {
        this.setState({
            memorySelection
        });
    }

    deselectTrace = () =>
    {
        this.props.selectTrace(null);
    }
}

export const TraceVisualisation = connect<StateProps, DispatchProps, {}>((state: GlobalState) => ({
    files: state.fileLoader.files,
    profile: state.profile.profile,
    selectedKernel: selectedKernel(state),
    selectedTrace: selectedTrace(state),
    selectedWarps: selectedWarps(state),
    traceSelection: state.trace.selectedTrace,
    warpSelection: state.trace.selectedWarps
}), {
    selectTrace: selectTrace,
    selectWarps: selectWarps,
    goToPage: push
})(TraceVisualisationComponent);
