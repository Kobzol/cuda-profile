import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {AccessType, Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';
import {Button, ListGroup, ListGroupItem, Card, CardHeader, CardBody} from 'reactstrap';
import {SourceLocation} from '../../../lib/profile/metadata';
import {SourceModal} from './source-modal/source-modal';
import styled from 'styled-components';
import {TraceHeader} from './trace-header';
import {TraceSelection} from '../../../lib/trace/selection';
import {contains} from 'ramda';
import {getFilename} from '../../../lib/util/string';
import {BadgeRead, BadgeWrite} from '../warp-access-ui';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWarps: Warp[];
    selectWarps(warps: Warp[]): void;
    selectTrace(selection: TraceSelection): void;
}

interface State
{
    blockFilter: Dim3;
    locationFilter: SourceLocation[];
    typeFilter: {
        read: boolean;
        write: boolean;
    };
    sourceModalOpened: boolean;
    activePanels: number[];
}

const Wrapper = styled(Card)`
  
`;
const BodyWrapper = styled(CardBody)`
  padding: 10px;
`;
const SourceLocationEntry = styled(ListGroupItem)`
  padding: 5px;
  font-size: 14px;
`;
const Section = styled.div`
  margin-top: 10px;
  :first-child {
    margin-top: 0;
  }
  
  h4 {
    margin: 0;
  }
`;
const FilterWrapper = styled.div`
  margin-bottom: 10px;
`;
const Row = styled.div`
  display: flex;
  align-items: center;
`;
const Label = styled.span`
  margin-right: 5px;
`;

export class WarpPanel extends PureComponent<Props, State>
{
    state: State = {
        blockFilter: { x: null, y: null, z: null },
        locationFilter: [],
        typeFilter: {
            read: true,
            write: true
        },
        sourceModalOpened: false,
        activePanels: []
    };

    render()
    {
        const warps = this.getFilteredWarps();
        return (
            <Wrapper>
                <CardHeader>Selected kernel</CardHeader>
                <BodyWrapper>
                    <TraceHeader
                        kernel={this.props.kernel}
                        trace={this.props.trace}
                        selectTrace={this.props.selectTrace} />
                    {this.props.kernel.metadata.source &&
                        <SourceModal
                            opened={this.state.sourceModalOpened}
                            kernel={this.props.kernel}
                            trace={this.props.trace}
                            locationFilter={this.state.locationFilter}
                            setLocationFilter={this.setLocationFilter}
                            onClose={this.closeSourceModal}/>
                    }
                    <Section>
                        <h4>Filters</h4>
                        {this.renderFilters(warps)}
                    </Section>
                    <Section>
                        <h4>Filtered access minimap</h4>
                        <WarpOverview
                            warps={warps}
                            selectedWarps={this.props.selectedWarps}
                            onWarpSelect={this.props.selectWarps} />
                    </Section>
                </BodyWrapper>
            </Wrapper>
        );
    }
    renderFilters = (warps: Warp[]): JSX.Element =>
    {
        const label = `${warps.length} accesses selected by filter (${this.props.trace.warps.length} total)`;
        const location = this.state.locationFilter.map(loc =>
            <SourceLocationEntry key={`${loc.file}:${loc.line}`}>
                {getFilename(loc.file)}:{loc.line}
            </SourceLocationEntry>
        );

        return (
            <>
                <FilterWrapper>
                    <Row>
                        <Label>Block:</Label>
                        <WarpFilter
                            filter={this.state.blockFilter}
                            onFilterChange={this.changeBlockFilter} />
                    </Row>
                    <Row>
                        <Label>Type:</Label>
                        <Row>
                            <input type='checkbox' checked={this.state.typeFilter.read}
                                   onChange={this.handleTypeReadChange} />
                            <BadgeRead>read</BadgeRead>
                        </Row>
                        <Row>
                            <input type='checkbox' checked={this.state.typeFilter.write}
                                   onChange={this.handleTypeWriteChange} />
                            <BadgeWrite>write</BadgeWrite>
                        </Row>
                    </Row>
                    {this.state.locationFilter.length > 0 &&
                    <div>
                        Source locations:
                        <ListGroup>{location}</ListGroup>
                    </div>}
                    {this.props.kernel.metadata.source &&
                    <Section>
                        <Button size='sm' onClick={this.openSourceModal}>Filter by source location</Button>
                    </Section>
                    }
                </FilterWrapper>
                <div>{label}</div>
                <Button onClick={this.resetFilters} color='danger'>Reset filter</Button>
            </>
        );
    }

    getFilteredWarps = (): Warp[] =>
    {
        const {x, y, z} = this.state.blockFilter;
        return this.props.trace.warps.filter(warp => {
            if (x !== null && warp.blockIdx.x !== x) return false;
            if (y !== null && warp.blockIdx.y !== y) return false;
            if (z !== null && warp.blockIdx.z !== z) return false;
            if (this.state.locationFilter.length > 0 && !this.testLocationFilter(warp)) return false;
            if (!this.state.typeFilter.read && warp.accessType === AccessType.Read) return false;
            if (!this.state.typeFilter.write && warp.accessType === AccessType.Write) return false;
            return true;
        });
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
            locationFilter: [],
            typeFilter: {
                read: true,
                write: true
            }
        }));
    }

    setLocationFilter = (locationFilter: SourceLocation[]) =>
    {
        this.setState(() => ({ locationFilter }));
    }
    testLocationFilter = (warp: Warp): boolean =>
    {
        const location: SourceLocation = { file: warp.location.file, line: warp.location.line };
        return contains(location, this.state.locationFilter);
    }
    handleTypeReadChange = (event: React.FormEvent<HTMLInputElement>) =>
    {
        const read = event.currentTarget.checked;
        this.setState(state => ({
            typeFilter: {
                ...state.typeFilter,
                read
            }
        }));
    }
    handleTypeWriteChange = (event: React.FormEvent<HTMLInputElement>) =>
    {
        const write = event.currentTarget.checked;
        this.setState(state => ({
            typeFilter: {
                ...state.typeFilter,
                write
            }
        }));
    }

    changeSourcePanelVisibility = (sourceModalOpened: boolean) =>
    {
        this.setState(() => ({ sourceModalOpened }));
    }
    closeSourceModal = () =>
    {
        this.changeSourcePanelVisibility(false);
    }
    openSourceModal = () =>
    {
        this.changeSourcePanelVisibility(true);
    }
}
