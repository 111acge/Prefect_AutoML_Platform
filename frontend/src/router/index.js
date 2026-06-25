// Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
// This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
// See LICENSE for details.

import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import DatasetsView from '@/views/DatasetsView.vue'
import RunsView from '@/views/RunsView.vue'
import RunDetailView from '@/views/RunDetailView.vue'
import CompareView from '@/views/CompareView.vue'
import PipelineView from '@/views/PipelineView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: DatasetsView,
  },
  {
    path: '/runs',
    name: 'runs',
    component: RunsView,
  },
  {
    path: '/runs/:id',
    name: 'run-detail',
    component: RunDetailView,
  },
  {
    path: '/compare',
    name: 'compare',
    component: CompareView,
  },
  {
    path: '/runs/:id/pipeline',
    name: 'pipeline',
    component: PipelineView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
