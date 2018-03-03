import React, {PureComponent} from 'react';
import {Warp} from '../../../../lib/profile/warp';
import {Button} from 'reactstrap';
import {WarpMiniature} from './warp-miniature';
import {SVGGrid} from '../../svg-grid/svg-grid';
import styled from 'styled-components';

interface Props
{
    warps: Warp[];
    selectedWarps: Warp[];
    onWarpSelect: (warp: Warp[]) => void;
}

interface State
{
    limit: number;
}

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

export class WarpOverview extends PureComponent<Props, State>
{
    state: State = {
        limit: 100
    };

    render()
    {
        const width = 320;
        const height = 200;
        const increaseLimit = this.state.limit < this.props.warps.length;
        return (
            <Wrapper>
                {this.props.warps.length === 0 ? 'No warps match the active filters' :
                    <SVGGrid width={width}
                             height={height}
                             rows={16}
                             cols={16}
                             renderItem={this.renderWarpMiniature}
                             {...{
                                 selectedWarps: this.props.selectedWarps,
                                 limit: this.state.limit
                             }} />
                }
                {increaseLimit && <Button onClick={this.increaseLimit} color='primary'>Show more warps</Button>}
            </Wrapper>
        );
    }
    renderWarpMiniature = (index: number, x: number, y: number, width: number, height: number): JSX.Element =>
    {
        if (index >= this.props.warps.length || index >= this.state.limit) return null;
        const warp = this.props.warps[index];
        const selected = this.props.selectedWarps.includes(warp);

        return (
            <WarpMiniature
                x={x}
                y={y}
                width={width}
                height={height}
                warp={warp}
                selected={selected}
                onClick={this.handleMiniatureClick} />
        );
    }

    handleMiniatureClick = (warp: Warp, ctrlPressed: boolean) =>
    {
        if (ctrlPressed)
        {
            this.handleSelectAdd(warp);
        }
        else this.handleSelect(warp);
    }

    increaseLimit = () =>
    {
        this.setState((state: State) => ({
            ...state,
            limit: state.limit + 100
        }));
    }

    handleSelect = (warp: Warp) =>
    {
        if (this.props.selectedWarps.includes(warp))
        {
            this.props.onWarpSelect(this.props.selectedWarps.filter(w => w !== warp));
        }
        else this.props.onWarpSelect([warp]);
    }
    handleSelectAdd = (warp: Warp) =>
    {
        if (this.props.selectedWarps.includes(warp))
        {
            this.props.onWarpSelect(this.props.selectedWarps.filter(w => w !== warp));
        }
        else this.props.onWarpSelect([...this.props.selectedWarps, warp]);
    }
}
