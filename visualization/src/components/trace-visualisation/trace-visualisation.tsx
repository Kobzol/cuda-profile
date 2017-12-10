import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {selectWarps, selectTrace, deselectWarp} from '../../lib/trace/actions';
import {GlobalState} from '../../lib/state/reducers';
import {TraceTimeline} from './trace-timeline/trace-timeline';
import {Profile} from '../../lib/profile/profile';
import {AddressRange, TraceSelection} from '../../lib/trace/selection';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {selectedWarps, selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {Warp} from '../../lib/profile/warp';
import {WarpList} from './warp-list/warp-list';
import {WarpAddressSelection} from '../../lib/trace/selection';
import {Routes} from '../../lib/nav/routes';
import {push} from 'react-router-redux';
import {WarpTimeline} from './warp-timeline/warp-timeline';
import {WarpDetail} from './warp-detail/warp-detail';
import {Button, Glyphicon} from 'react-bootstrap';
import * as moment from 'moment';
import {Action} from 'typescript-fsa';
import * as _ from 'lodash';

import './trace-visualisation.scss';
import {WarpPanel} from './warp-panel/warp-panel';

export const selectAllWarpAccesses = (warp: Warp) =>
{
    return (dispatch: (action: Action<Warp[]>) => void, getState: () => GlobalState) => {
        const state = getState();
        const trace = selectedTrace(state);
        const warps = trace.warps.filter(w => w.id === warp.id && _.isEqual(warp.blockIdx, w.blockIdx));

        dispatch(selectWarps(warps));
    };
};

interface StateProps
{
    files: TraceFile[];
    profile: Profile;
    selectedKernel: Kernel;
    selectedTrace: Trace;
    selectedWarps: Warp[];
    traceSelection: TraceSelection;
}
interface DispatchProps
{
    selectTrace: (selection: TraceSelection) => {};
    selectWarps: (warps: Warp[]) => {};
    deselectWarp: (warp: Warp) => {};
    goToPage: (page: string) => {};
    selectAllWarpAccesses: (warp: Warp) => {};
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
            <div className='trace-content-wrapper'>
                <div className='warp-panel'>
                    <WarpPanel
                        kernel={kernel}
                        trace={trace}
                        selectWarps={this.props.selectWarps}
                        selectedWarps={warps} />
                </div>
                <div className='trace-wrapper'>
                    <WarpList
                        trace={trace}
                        warps={warps}
                        selectRange={(range) => this.setState({
                            rangeSelections: range === null ? [] : [range]
                        })}
                        memorySelection={this.state.memorySelection}
                        deselect={this.props.deselectWarp}
                        selectAllWarpAccesses={this.props.selectAllWarpAccesses} />
                    <WarpDetail
                        trace={trace}
                        warps={warps}
                        rangeSelections={this.state.rangeSelections}
                        onMemorySelect={this.setMemorySelection} />
                </div>
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
    traceSelection: state.trace.selectedTrace
}), {
    selectTrace: selectTrace,
    selectWarps: selectWarps,
    deselectWarp: deselectWarp,
    selectAllWarpAccesses: selectAllWarpAccesses,
    goToPage: push
})(TraceVisualisationComponent);
