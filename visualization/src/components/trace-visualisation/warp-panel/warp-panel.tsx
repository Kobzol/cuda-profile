import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter/warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';
import {Button, ListGroup, ListGroupItem, Panel, PanelGroup} from 'react-bootstrap';
import {SourceView} from '../source-view/source-view';
import {SourceLocation} from '../../../lib/profile/metadata';
import _ from 'lodash';

import style from './warp-panel.scss';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWarps: Warp[];
    selectWarps: (warps: Warp[]) => void;
}

interface State
{
    blockFilter: Dim3;
    locationFilter: SourceLocation[];
    sourcePanelOpened: boolean;
    activePanels: number[];
}

export class WarpPanel extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            blockFilter: { x: null, y: null, z: null },
            locationFilter: [],
            sourcePanelOpened: false,
            activePanels: []
        };
    }

    render()
    {
        const warps = this.getFilteredWarps();
        return (
            <Panel header='Warps' className={style.warpsWrapper} bsStyle='primary'>
                {this.state.sourcePanelOpened &&
                <SourceView content={this.props.kernel.metadata.source.content}
                            file={this.props.kernel.metadata.source.file}
                            warps={this.props.trace.warps}
                            locationFilter={this.state.locationFilter}
                            setLocationFilter={this.setLocationFilter}
                            onClose={() => this.changeSourcePanelVisibility(false)} />
                }
                <PanelGroup>
                    <Panel collapsible header='Active filters' className={style.panel}>
                            {this.isFilterActive() ? this.renderFilter(warps) :
                                `No filters (${this.props.trace.warps.length} total warps)`}
                    </Panel>
                    <Panel collapsible header='Filter by block index' className={style.panel}>
                        <WarpFilter
                            filter={this.state.blockFilter}
                            onFilterChange={this.changeBlockFilter} />
                    </Panel>
                    <Panel collapsible={false}
                           onClick={() => this.changeSourcePanelVisibility(!this.state.sourcePanelOpened)}
                           header={<h4>Filter by source code</h4>}
                           className={style.panel} />
                </PanelGroup>
                <div>
                    <h5>Warp map</h5>
                    <WarpOverview
                        warps={warps}
                        selectedWarps={this.props.selectedWarps}
                        onWarpSelect={this.props.selectWarps} />
                </div>
            </Panel>
        );
    }
    renderFilter = (warps: Warp[]): JSX.Element =>
    {
        const label = `${warps.length} selected by filter (${this.props.trace.warps.length} total)`;

        const {x, y, z} = this.state.blockFilter;
        const dim = `${z || 'z'}.${y || 'y'}.${x || 'x'}`;
        const location = this.state.locationFilter.map(loc =>
            <ListGroupItem>
                {loc.file}:{loc.line}
            </ListGroupItem>
        );

        return (
            <div>
                <div>Block index: {dim}</div>
                {this.state.locationFilter.length > 0 &&
                <div>
                    Source locations:
                    <ListGroup>{location}</ListGroup>
                </div>}
                <div>{label}</div>
                <Button onClick={this.resetFilters} bsStyle='danger'>Reset filter</Button>
            </div>
        );
    }

    getFilteredWarps = (): Warp[] =>
    {
        const {x, y, z} = this.state.blockFilter;
        if (!this.isFilterActive()) return this.props.trace.warps;

        return this.props.trace.warps.filter(warp => {
            if (x !== null && warp.blockIdx.x !== x) return false;
            if (y !== null && warp.blockIdx.y !== y) return false;
            if (z !== null && warp.blockIdx.z !== z) return false;
            if (this.state.locationFilter.length > 0 && !this.testLocationFilter(warp)) return false;
            return true;
        });
    }

    isFilterActive = (): boolean =>
    {
        return (
            this.state.blockFilter.x !== null ||
            this.state.blockFilter.y !== null ||
            this.state.blockFilter.z !== null ||
            this.state.locationFilter.length > 0
        );
    }

    changeBlockFilter = (blockFilter: Dim3) =>
    {
        this.setState(() => ({
            blockFilter
        }));
    }
    resetFilters = () =>
    {
        this.setState(() => ({
            blockFilter: { x: null, y: null, z: null },
            locationFilter: []
        }));
    }

    setLocationFilter = (locationFilter: SourceLocation[]) =>
    {
        this.setState(() => ({ locationFilter }));
    }
    testLocationFilter = (warp: Warp): boolean =>
    {
        const location: SourceLocation = { file: warp.location.file, line: warp.location.line };
        return _.some(this.state.locationFilter, location);
    }

    changeSourcePanelVisibility = (show: boolean) =>
    {
        this.setState(() => ({
            sourcePanelOpened: show
        }));
    }
}
