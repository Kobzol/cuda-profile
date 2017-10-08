import React, {PureComponent} from 'react';
import {MemoryAccess, MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {Grid} from './grid/grid';
import {
    GridData, GridSelection, DataSelection, GridBounds,
    createBlockSelector, createBoundsSelector
} from './grid/grid-data';
import {GridNavigator} from './grid/grid-navigator';
import {Selector} from 'reselect';
import * as _ from 'lodash';

interface Props
{
    id: string;
    accessGroup: MemoryAccessGroup;
}

interface State
{
    selection: GridSelection;
    calculateData: Selector<MemoryAccessGroup, GridData<MemoryAccess>>;
    calculateBounds: Selector<DataSelection<MemoryAccess>, GridBounds>;
}

const dimensions = [8, 16, 32, 64];

export class ThreadSelection extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            selection: {
                z: 0,
                y: 0,
                x: 0,
                width: 32,
                height: 32
            },
            calculateData: createBlockSelector(),
            calculateBounds: createBoundsSelector()
        };
    }

    render()
    {
        const data = this.state.calculateData(this.props.accessGroup);
        const selection = this.state.selection;
        const bounds = this.state.calculateBounds({data, selection});

        return (
            <div>
                <GridNavigator
                    selection={selection}
                    onSelectionChanged={this.handleSelectionChange}
                    dimensions={dimensions}
                    bounds={bounds} />
                <Grid
                    id={this.props.id}
                    data={data}
                    selection={selection}
                    canvasDimensions={{ width: 1600, height: 900 }} />
            </div>
        );
    }

    handleSelectionChange = (selection: GridSelection) =>
    {
        selection = this.normalizeSelection(
            this.state.calculateBounds({
                data: this.state.calculateData(this.props.accessGroup),
                selection: this.state.selection
            }),
            selection
        );
        this.setState(() => ({ selection }));
    }

    normalizeSelection = (bounds: GridBounds, selection: GridSelection): GridSelection =>
    {
        let {z, y, x} = selection;
        if (_.sortedIndexOf(bounds.z, z) === -1) z = 0;
        if (_.sortedIndexOf(bounds.z, z) === -1) z = 0;
        
        return selection;
    }
}
