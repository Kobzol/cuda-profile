import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter/warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';
import {Button, Panel} from 'react-bootstrap';
import {SourceView} from '../source-view/source-view';
import {SourceLocation} from '../../../lib/profile/metadata';

import './warp-panel.scss';
import * as _ from 'lodash';

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
}

export class WarpPanel extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            blockFilter: { x: null, y: null, z: null },
            locationFilter: [],
            sourcePanelOpened: false
        };
    }

    render()
    {
        const warps = this.getFilteredWarps();
        return (
            <div className='warps-wrapper'>
                {this.state.sourcePanelOpened &&
                <SourceView content={this.props.kernel.metadata.source.content}
                            file={this.props.kernel.metadata.source.file}
                            warps={this.props.trace.warps}
                            locationFilter={this.state.locationFilter}
                            setLocationFilter={this.setLocationFilter}
                            onClose={() => this.changeSourcePanelVisibility(false)} />
                }
                <h3>Warps</h3>
                {this.isFilterActive() ? this.renderFilter(warps) :
                    `${this.props.trace.warps.length} total`}
                <WarpFilter
                    label='Block'
                    filter={this.state.blockFilter}
                    onFilterChange={this.changeBlockFilter} />
                <Button onClick={() => this.changeSourcePanelVisibility(true)}>Filter by source code</Button>
                <WarpOverview
                    warps={warps}
                    selectedWarps={this.props.selectedWarps}
                    onWarpSelect={this.props.selectWarps} />
            </div>
        );
    }
    renderFilter = (warps: Warp[]): JSX.Element =>
    {
        const label = `${warps.length} filtered (${this.props.trace.warps.length} total)`;

        const {x, y, z} = this.state.blockFilter;
        const dim = `${z === null ? 'z' : z}.${y === null ? 'y' : y}.${x === null ? 'x' : x}`;
        const location = this.state.locationFilter.map(loc => `${loc.file}:${loc.line}`).join(', ');

        return (
            <Panel header={'Active filters'}>
                <div>Block index: {dim}</div>
                {this.state.locationFilter.length > 0 && <div>Source locations: {location}</div>}
                <div>{label}</div>
                <Button onClick={this.resetFilters}>Reset filter</Button>
            </Panel>
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

    setLocationFilter = (location: SourceLocation) =>
    {
        this.setState(() => ({
            locationFilter: [location]
        }));
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
