# Motor de ajedrez toma de deciciones

import chess
from stockfish import Stockfish

class Brain:
    def __init__(self):
        self.sf = Stockfish()
        self.ucim=[]
        self.tablero = chess.Board()
        print(self.tablero)
    def legal(self,mv):
        mov=self.to_move(mv)
        return (mov in self.tablero.legal_moves)
    
    def mover_uci(self,mv):
        self.ucim.append(mv)
        self.sf.set_position(self.ucim)
        self.tablero.push_uci(mv)
        print(self.tablero)
    def auto(self):
        mv=self.sf.get_best_move()  
        print(mv)
        return(mv)
    def to_move(self,mv):
        return chess.Move.from_uci(mv)
    def is_over(self):
        return self.tablero.is_game_over()
    def capturo(self,mv):
        if self.tablero.is_capture(mv):
            if self.tablero.is_en_passant(mv):
                return (True,True)
            else:
                return (True,False)
        else:
            return (False,False)
        return 
    def enroque(self,mv):
        if self.tablero.is_kingside_castling(mv):
            return(True,True)
        elif self.tablero.is_queenside_castling(mv):
            return(True,False)
        else:
            return(False,False)
        
        
    
##tablero = chess.Board()
##print(tablero)
