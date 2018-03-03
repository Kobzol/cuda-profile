import React, {ChangeEvent, PureComponent} from 'react';
import {Dim3} from '../../../../lib/profile/dim3';
import styled from 'styled-components';
import Input from 'reactstrap/lib/Input';

interface Props
{
    filter: Dim3;
    onFilterChange: (filter: Dim3) => void;
}

const FilterWrapper = styled.div`
  display: flex;
  align-items: center;

  input[type=number] {
    width: 50px;
  }
`;
const AxisWrapper = styled.div`
  margin-right: 5px;
`;
const AxisLabel = styled.div`
  text-align: center;
`;
const AxisInput = styled(Input)`
  padding: 0;
`;

export class WarpFilter extends PureComponent<Props>
{
    render()
    {
        return (
            <FilterWrapper>
                {this.renderDimension('z')}
                {this.renderDimension('y')}
                {this.renderDimension('x')}
            </FilterWrapper>
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
            <AxisWrapper>
                <AxisLabel>{dim}</AxisLabel>
                <AxisInput type='number' name={dim}
                       min='0' value={this.props.filter[dim] === null ? '' : this.props.filter[dim]}
                       autoComplete='false' onChange={this.handleChange} />
            </AxisWrapper>
        );
    }
}
