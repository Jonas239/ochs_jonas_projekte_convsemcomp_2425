from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
from chess import Move
from stockfish import Stockfish
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

app = Flask(__name__, static_folder="static")
CORS(app)

STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"
stockfish = Stockfish(STOCKFISH_PATH)

stockfish.update_engine_parameters({
    "Skill Level": 15,
    "UCI_LimitStrength": True,  
    "UCI_Elo": 2600
})

board = chess.Board()

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    model="llama-3.2-3b-instruct",
    temperature=0.7,
    streaming=True,
    max_tokens=300
)

analysis_template = PromptTemplate(
    input_variables=["fen", "stockfish_output"],
    template=(
        "You are a chess coach analyzing a game. I am playing the white pieces. "
        "The current chess board state is in FEN: {fen}. "
        "Here is Stockfish's detailed analysis: {stockfish_output}. Provide a natural language explanation for this position."
        "Only rely on the information provided by Stockfish. Do not repeat the board position in fen"
    ),
)

chat_template = PromptTemplate(
    input_variables=["fen", "chat", "best_move", "stockfish_output"],
    template=(
        "I am playing the white pieces. "
        "The current chess board state is in FEN: {fen} and Stockfish's evaluation is {stockfish_output}."
        "You are having a conversation with a chess player. Provide a response to their message: {chat}."
        "If the player asks for a move, provide a move suggestion for white based on Stockfish's evaluation: {best_move}."
        "Only rely on the information provided by Stockfish."
    ),
)

@app.route("/state", methods=["GET"])
def get_board_state():
    """Return the current board state in FEN format."""
    return jsonify({"fen": board.fen()})

@app.route("/move", methods=["POST"])
def make_move():
    """Process the player's move."""
    global board
    data = request.json
    move = data.get("move", "")

    try:
        uci_move = chess.Move.from_uci(move)
        if uci_move in board.legal_moves:
            board.push(uci_move)

            if board.is_checkmate():
                return jsonify({
                    "status": "checkmate",
                    "winner": "player",
                    "message": "Checkmate! Congratulations, you won!",
                    "fen": board.fen()
                })
            if board.is_stalemate():
                return jsonify({
                    "status": "stalemate",
                    "message": "It's a stalemate. The game is a draw.",
                    "fen": board.fen()
                })
            if board.is_check():
                return jsonify({
                    "status": "check",
                    "message": "Your move resulted in a check!",
                    "fen": board.fen()
                })

            stockfish.set_fen_position(board.fen())
            stockfish_move = stockfish.get_best_move()
            if stockfish_move:
                board.push(chess.Move.from_uci(stockfish_move))

                if board.is_checkmate():
                    return jsonify({
                        "status": "checkmate",
                        "winner": "stockfish",
                        "message": "Checkmate! Stockfish wins!",
                        "fen": board.fen()
                    })
                if board.is_stalemate():
                    return jsonify({
                        "status": "stalemate",
                        "message": "It's a stalemate. The game is a draw.",
                        "fen": board.fen()
                    })
                if board.is_check():
                    return jsonify({
                        "status": "check",
                        "message": "Stockfish's move resulted in a check!",
                        "fen": board.fen()
                    })

            return jsonify({"fen": board.fen()})
        else:
            return jsonify({
                "error": "Illegal move",
                "message": "The move you attempted is not legal.",
                "fen": board.fen()
            })
    except ValueError:
        return jsonify({
            "error": "Invalid move format",
            "message": "The move format is invalid. Please try again.",
            "fen": board.fen()
        })

@app.route("/analyze", methods=["POST"])
def analyze_board():
    """Provide detailed analysis of the board."""
    global board
    stockfish.set_fen_position(board.fen())
    evaluation = stockfish.get_evaluation()
    prompt = analysis_template.format(
        fen=board.fen(),
        stockfish_output=evaluation,
    )
    explanation = llm.invoke([{"role": "user", "content": prompt}]).content
    return jsonify({"fen": board.fen(), "evaluation": evaluation, "explanation": explanation})

@app.route("/chat", methods=["POST"])
def chat_with_board():
    global board
    stockfish.set_fen_position(board.fen())
    evaluation = stockfish.get_evaluation()
    best_move = stockfish.get_best_move()
    prompt = chat_template.format(
        fen=board.fen(),
        chat=request.json.get("chat", ""),
        best_move=best_move,
        stockfish_output=evaluation,
    )
    response = llm.invoke([{"role": "user", "content": prompt}]).content
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)