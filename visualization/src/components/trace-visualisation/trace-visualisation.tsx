import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {selectWarps, selectTrace, deselectWarp, clearWarpSelection} from '../../lib/trace/actions';
import {GlobalState} from '../../lib/state/reducers';
import {KernelTimeline} from './kernel-timeline';
import {Profile} from '../../lib/profile/profile';
import {AddressRange, TraceSelection, WarpAccess} from '../../lib/trace/selection';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {selectedWarps, selectedKernel, selectedTrace} from '../../lib/trace/reducer';
import {Warp} from '../../lib/profile/warp';
import {WarpList} from './warp-list/warp-list';
import {Routes} from '../../lib/nav/routes';
import {push} from 'react-router-redux';
import {WarpDetail} from './warp-detail/warp-detail';
import {WarpPanel} from './warp-panel/warp-panel';
import {Action} from 'typescript-fsa';
import {withRouter} from 'react-router';
import styled from 'styled-components';
import {equals, without} from 'ramda';

export const selectAllWarpAccesses = (warp: Warp) =>
{
    return (dispatch: (action: Action<Warp[]>) => void, getState: () => GlobalState) => {
        const state = getState();
        const trace = selectedTrace(state);
        const warps = trace.warps.filter(w => w.id === warp.id && equals(warp.blockIdx, w.blockIdx));

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
    selectTrace(selection: TraceSelection): void;
    selectWarps(warps: Warp[]): void;
    deselectWarp(warp: Warp): void;
    clearWarpSelection(): void;
    goToPage(page: string): void;
    selectAllWarpAccesses(warp: Warp): void;
}

type Props = StateProps & DispatchProps;

interface State
{
    selectedAccesses: WarpAccess[];
    memorySelection: AddressRange[];
}


const Row = styled.div`
  display: flex;
`;
const TimelineWrapper = styled.div`
  flex-grow: 1;
`;
const Column = styled.div`
  display: flex;
  flex-direction: column;
`;
const LeftColumn = Column.extend`
  width: 350px;
  margin-right: 10px;
`;
const RightColumn = Column.extend`
  flex-grow: 1;
`;

class TraceVisualisationComponent extends PureComponent<Props, State>
{
    state: State = {
        selectedAccesses: [],
        memorySelection: []
    };

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
        return (
            <Row>
                {this.props.selectedTrace === null && this.renderKernelTimeline()}
                {this.renderLeftColumn()}
                {this.renderRightColumn()}
            </Row>
        );
    }
    renderKernelTimeline = (): JSX.Element =>
    {
        return (
            <TimelineWrapper>
                <KernelTimeline
                    profile={this.props.profile}
                    selection={this.props.traceSelection}
                    selectTrace={this.props.selectTrace} />
            </TimelineWrapper>
        );
    }
    renderLeftColumn = (): JSX.Element =>
    {
        if (this.props.selectedTrace === null) return null;

        return (
            <LeftColumn>
                <WarpPanel
                    kernel={this.props.selectedKernel}
                    trace={this.props.selectedTrace}
                    selectWarps={this.props.selectWarps}
                    selectedWarps={this.props.selectedWarps}
                    selectTrace={this.props.selectTrace} />
            </LeftColumn>
        );
    }
    renderRightColumn = (): JSX.Element =>
    {
        if (this.props.selectedTrace === null) return null;

        return (
            <RightColumn>
                <WarpList
                    trace={this.props.selectedTrace}
                    warps={this.props.selectedWarps}
                    memorySelection={this.state.memorySelection}
                    selectedAccesses={this.state.selectedAccesses}
                    onAccessSelectionChange={this.handleAccessSelectChange}
                    onDeselect={this.deselectWarp}
                    onClearSelection={this.clearWarpSelection}
                    onSelectAllWarpAccesses={this.props.selectAllWarpAccesses} />
                <WarpDetail
                    trace={this.props.selectedTrace}
                    warps={this.props.selectedWarps}
                    selectedAccesses={this.state.selectedAccesses}
                    onMemorySelect={this.setMemorySelection} />
            </RightColumn>
        );
    }

    handleAccessSelectChange = (access: WarpAccess, active: boolean) =>
    {
        this.setState(state => ({
            selectedAccesses: active ? [...state.selectedAccesses, access] : without([access], state.selectedAccesses)
        }));
    }
    deselectWarp = (warp: Warp) =>
    {
        this.setState(state => ({
            selectedAccesses: state.selectedAccesses.filter(w => w.warp.index !== warp.index)
        }));
        this.props.deselectWarp(warp);
    }
    clearWarpSelection = () =>
    {
        this.setState(() => ({
            selectedAccesses: []
        }));
        this.props.clearWarpSelection();
    }

    setMemorySelection = (memorySelection: AddressRange[]) =>
    {
        this.setState({ memorySelection });
    }
}

export const TraceVisualisation = withRouter(connect<StateProps, DispatchProps>((state: GlobalState) => ({
    files: state.fileLoader.files,
    profile: state.profile.profile,
    selectedKernel: selectedKernel(state),
    selectedTrace: selectedTrace(state),
    selectedWarps: selectedWarps(state),
    traceSelection: state.trace.selectedTrace
}), {
    selectTrace,
    selectWarps,
    deselectWarp,
    clearWarpSelection,
    selectAllWarpAccesses,
    goToPage: push
})(TraceVisualisationComponent));
