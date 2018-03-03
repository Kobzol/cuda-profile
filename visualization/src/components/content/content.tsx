import React, {PureComponent} from 'react';
import {TraceVisualisation} from '../trace-visualisation/trace-visualisation';
import {TraceLoader} from '../trace-loader/trace-loader';
import {Routes} from '../../lib/nav/routes';
import {Route, withRouter} from 'react-router';
import styled from 'styled-components';
import {Nav, NavItem, NavLink, Navbar, NavbarBrand} from 'reactstrap';

import './content.scss';

const Root = styled.div`
  width: 100%;
  height: 100%;
  margin: 0 auto;
  word-wrap: break-word;
`;
const ContentWrapper = styled.div`
  padding: 10px;
`;

class ContentComponent extends PureComponent
{
    render()
    {
        return (
            <Root>
                <Navbar dark color='dark' expand='md'>
                    <NavbarBrand href='/'>Trace visualisation</NavbarBrand>
                    <Nav navbar>
                        <NavItem><NavLink href={Routes.Root}>Load trace</NavLink></NavItem>
                    </Nav>
                </Navbar>
                <ContentWrapper>
                    <Route path={Routes.Root} exact component={TraceLoader} />
                    <Route path={Routes.TraceVisualisation} component={TraceVisualisation} />
                </ContentWrapper>
            </Root>
        );
    }
}

export const Content = withRouter(ContentComponent);
