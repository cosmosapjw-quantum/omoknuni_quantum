// src/python/bindings.cpp
// Simplified Python bindings for game logic only
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>

#include "core/igamestate.h"
#include "core/game_export.h"

#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"

namespace py = pybind11;

namespace alphazero {
namespace python {

// Convert C++ tensor representation to Python numpy arrays
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
    if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
        return py::array_t<float>();
    }
    
    size_t channels = tensor.size();
    size_t height = tensor[0].size();
    size_t width = tensor[0][0].size();
    
    py::array_t<float> array({channels, height, width});
    py::buffer_info info = array.request();
    float* data = static_cast<float*>(info.ptr);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                data[c * height * width + h * width + w] = tensor[c][h][w];
            }
        }
    }
    
    return array;
}

// Module definition
PYBIND11_MODULE(alphazero_py, m) {
    m.doc() = "AlphaZero Python bindings - Game Logic Only";
    
    // Game types
    py::enum_<core::GameType>(m, "GameType")
        .value("UNKNOWN", core::GameType::UNKNOWN)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .value("GOMOKU", core::GameType::GOMOKU)
        .export_values();
    
    // Game result
    py::enum_<core::GameResult>(m, "GameResult")
        .value("ONGOING", core::GameResult::ONGOING)
        .value("WIN_PLAYER1", core::GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", core::GameResult::WIN_PLAYER2)
        .value("DRAW", core::GameResult::DRAW)
        .export_values();
    
    // Game interface
    py::class_<core::IGameState>(m, "IGameState")
        .def("get_legal_moves", &core::IGameState::getLegalMoves)
        .def("is_legal_move", &core::IGameState::isLegalMove)
        .def("make_move", &core::IGameState::makeMove)
        .def("undo_move", &core::IGameState::undoMove)
        .def("is_terminal", &core::IGameState::isTerminal)
        .def("get_game_result", &core::IGameState::getGameResult)
        .def("get_current_player", &core::IGameState::getCurrentPlayer)
        .def("get_board_size", &core::IGameState::getBoardSize)
        .def("get_action_space_size", &core::IGameState::getActionSpaceSize)
        .def("get_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getTensorRepresentation());
        })
        .def("get_enhanced_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getEnhancedTensorRepresentation());
        })
        .def("get_hash", &core::IGameState::getHash)
        .def("action_to_string", &core::IGameState::actionToString)
        .def("string_to_action", &core::IGameState::stringToAction)
        .def("to_string", &core::IGameState::toString)
        .def("get_move_history", &core::IGameState::getMoveHistory)
        .def("clone", &core::IGameState::clone);
    
    // Game factory
    m.def("create_game", [](core::GameType type) {
        return core::GameFactory::createGame(type);
    });
    
    m.def("create_game_from_moves", [](core::GameType type, const std::string& moves) {
        return core::GameFactory::createGameFromMoves(type, moves);
    });
    
    // Game serialization
    m.def("save_game", &core::GameSerializer::saveGame);
    m.def("load_game", &core::GameSerializer::loadGame);
    m.def("serialize_game", &core::GameSerializer::serializeGame);
    m.def("deserialize_game", &core::GameSerializer::deserializeGame);
    
    // Utility functions
    m.def("game_type_to_string", &core::gameTypeToString);
    m.def("string_to_game_type", &core::stringToGameType);
    
    // Game-specific classes
    py::class_<games::chess::ChessState, core::IGameState>(m, "ChessState")
        .def(py::init<>())
        .def(py::init<const games::chess::ChessState&>());

    py::class_<games::go::GoState, core::IGameState>(m, "GoState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 19);

    py::class_<games::gomoku::GomokuState, core::IGameState>(m, "GomokuState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 15);
}

} // namespace python
} // namespace alphazero