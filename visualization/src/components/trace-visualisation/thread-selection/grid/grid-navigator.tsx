import React, {ChangeEvent, PureComponent} from 'react';
import {GridBounds, GridSelection} from './grid-data';

interface Props
{
    dimensions: number[];
    selection: GridSelection;
    onSelectionChanged: (selection: GridSelection) => void;
    bounds: GridBounds;
}

export class GridNavigator extends PureComponent<Props>
{
    render()
    {
        return (
            <div>
                {this.renderNumberSelect(this.props.dimensions, this.props.selection.width, 'width')}
                {this.renderNumberSelect(this.props.dimensions, this.props.selection.height, 'height')}
                {this.renderNumberSelect(this.props.bounds.z, this.props.selection.z, 'z')}
                {this.renderNumberSelect(this.props.bounds.y, this.props.selection.y, 'y')}
                {this.renderNumberSelect(this.props.bounds.x, this.props.selection.x, 'x')}
            </div>
        );
    }

    renderNumberSelect = (dimensions: number[], value: number, attribute: string): JSX.Element =>
    {
        return (
            <select
                onChange={(event: ChangeEvent<HTMLSelectElement>) => {
                    this.changeSelection(attribute, parseInt(event.target.value, 10));
                }}
                value={value}>
                {dimensions.map(dim =>
                    <option
                        key={dim}
                        value={dim}>
                        {dim}
                    </option>
                )}
            </select>
        );
    }

    changeSelection = (attribute: string, value: number) =>
    {
        this.props.onSelectionChanged({
            ...this.props.selection,
            [attribute]: value
        });
    }
}
