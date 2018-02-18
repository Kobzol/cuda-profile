import React, {ChangeEvent, PureComponent} from 'react';
import {Dim3} from '../../../../lib/profile/dim3';

import style from './warp-filter.scss';

interface Props
{
    filter: Dim3;
    onFilterChange: (filter: Dim3) => void;
}

export class WarpFilter extends PureComponent<Props>
{
    render()
    {
        return (
            <div className={style.warpFilter}>
                {this.renderDimension('z')}
                {this.renderDimension('y')}
                {this.renderDimension('x')}
            </div>
        );
    }

    handleChange = (event: ChangeEvent<HTMLInputElement>) =>
    {
        const filter = {...this.props.filter};
        const num = parseInt(event.target.value, 10);

        filter[event.target.name] = isNaN(num) ? null : num;
        this.props.onFilterChange(filter);
    }

    renderDimension = (dim: string) =>
    {
        return (
            <div className={style.dimension}>
                <span>{dim}</span>
                <input type='number' name={dim}
                       min='0' value={this.props.filter[dim] === null ? '' : this.props.filter[dim]}
                       autoComplete='false' onChange={this.handleChange} />
            </div>
        );
    }
}