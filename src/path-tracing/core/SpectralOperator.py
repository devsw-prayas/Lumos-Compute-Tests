from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from core.SpectralState import SpectralState


class SpectralOperator:

    def __init__(self, basis: GHGSFMultiLobeBasis):
        self.m_basis  = basis
        self.m_matrix = None

    def buildMatrix(self):
        raise NotImplementedError

    def apply(self, state: SpectralState):

        if self.m_matrix is None:
            self.buildMatrix()

        state.m_coeffs = self.m_matrix @ state.m_coeffs

    def compose(self, other):

        if self.m_matrix is None:
            self.buildMatrix()

        if other.m_matrix is None:
            other.buildMatrix()

        composed = SpectralOperator(self.m_basis)
        composed.m_matrix = self.m_matrix @ other.m_matrix
        return composed

